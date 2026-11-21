
pipeline {
    agent any

    options {
        skipDefaultCheckout(true)
        disableConcurrentBuilds()
        timestamps()
    }

    triggers {
        pollSCM('H/2 * * * *')
    }

    stages {
        stage('Checkout') {
            steps {
                checkout scm
                script {
                    def commit = bat(
                        returnStdout: true,
                        script: '@git rev-parse --short=12 HEAD'
                    ).trim()

                    env.APP_TAG = "git-${commit}-${env.BUILD_NUMBER}"
                    env.CI_PROJECT = "fake-news-ci-${env.BUILD_NUMBER}"
                }
            }
        }

        stage('Build Docker Image') {
            steps {
                bat 'docker version'
                bat 'docker compose version'
                bat 'docker build -t fake-news-detector-web:%APP_TAG% .'
            }
        }

        stage('Test Model and Database') {
            environment {
                POSTGRES_PASSWORD = credentials('fake-news-db-password')
            }

            steps {
                bat '''
                    docker compose -p %CI_PROJECT% up -d --wait --wait-timeout 180 db
                '''

                bat '''
                    docker compose -p %CI_PROJECT% run --rm --no-deps --pull never -T web python -m unittest discover -s tests -v
                '''
            }
        }

        stage('Deploy Application') {
            environment {
                POSTGRES_PASSWORD = credentials('fake-news-db-password')
            }

            steps {
                bat '''
                    docker compose -p fake-news-detector up -d --no-build --pull never --wait --wait-timeout 180
                '''
            }
        }

        stage('Verify Deployment') {
            environment {
                POSTGRES_PASSWORD = credentials('fake-news-db-password')
            }

            steps {
                bat '''
                    docker compose -p fake-news-detector exec -T web python -c "from database import recent_predictions; recent_predictions(1); print('Web to PostgreSQL OK')"
                '''

                bat 'docker compose -p fake-news-detector ps'
            }
        }
    }

    post {
        always {
            script {
                if (env.CI_PROJECT?.trim()) {
                    withCredentials([
                        string(
                            credentialsId: 'fake-news-db-password',
                            variable: 'POSTGRES_PASSWORD'
                        )
                    ]) {
                        bat(
                            returnStatus: true,
                            script: 'docker compose -p %CI_PROJECT% down --volumes --remove-orphans'
                        )
                    }
                }
            }
        }

        success {
            echo 'CI/CD Pipeline completed successfully!'
        }

        failure {
            echo 'CI/CD Pipeline failed. Check Console Output.'
        }
    }
}
